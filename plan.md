# Plan inicial para la aplicación de directorio de exalumnos

## Objetivo
Construir una aplicación web responsive que permita a los exalumnos gestionar su información de contacto y estado personal de forma segura, con control total alojado en tus propios servidores.

## Estado actual (Iteración 1)
- Proyecto Django inicializado con autenticación por correo electrónico y panel de administración listo para crear cuentas.
- Modelo `AlumniProfile` y formulario web para que cada usuario actualice sus datos familiares y laborales.
- Registro de auditoría básico que captura los cambios relevantes y los muestra en la vista de perfil.
- Documentación de instalación rápida y variables de entorno clave en `README.md`.


## Resumen de requisitos confirmados
- **Acceso**: Sólo exalumnos dados de alta manualmente (nombre + email). Cada usuario puede editar únicamente su registro.
- **Consentimiento y privacidad**: Se solicitará consentimiento explícito. Datos visibles únicamente para la comunidad del curso.
- **Seguridad**: Despliegue sobre servidor propio con Nginx + SSL. Conexión HTTPS suficiente.
- **Autogestión**: Posibilidad de auto-registro sujeto a aprobación manual.
- **Experiencia de usuario**: Web responsive, sin envíos automáticos de emails por ahora. Campo de texto libre para describir el trabajo. Sin adjuntos iniciales.
- **Datos**: Campos clave sobre situación familiar (padres vivos, estado civil, hijos), trabajo, contacto y otros datos biográficos.
- **Exportación**: Exportar listados a CSV (y potencialmente PDF) para compartir offline.
- **Auditoría**: Registrar historial de cambios por usuario.
- **Escalabilidad**: 40 usuarios iniciales, diseño preparado hasta ~500.
- **Integraciones**: Sin integración inmediata con WhatsApp, se mantiene opción para futuro.

## Recomendaciones de arquitectura
1. **Backend**: Django + Django REST Framework (para futuras APIs). Aprovechar el panel de administración para gestión interna.
2. **Base de datos**: PostgreSQL (soporta JSON, transacciones robustas y escalabilidad). SQLite aceptable para desarrollo local.
3. **Autenticación**:
   - Crear usuarios manualmente desde el admin o vía importación CSV.
   - Implementar flujo de registro con token de invitación y aprobación manual.
   - Usar `django-allauth` sólo si en el futuro se requieren autenticaciones externas; por ahora bastaría la autenticación clásica.
4. **Autorización**: Modelo `Profile` con relación uno a uno a `User` y permisos personalizados para evitar modificaciones cruzadas.
5. **Auditoría**: Integrar `django-simple-history` o `django-reversion` para historial de cambios en el modelo `Profile`.
6. **Front-end**: Formularios Django (o Django Crispy Forms) con plantilla responsive (Bootstrap 5). Permite iterar rápido sin SPA.
7. **Exportaciones**:
   - Generación de CSV nativa con vistas protegidas.
   - Para PDF: `WeasyPrint` o `xhtml2pdf` si se requiere; recomendación diferir hasta validar necesidad.
8. **Despliegue**:
   - Servidor Ubuntu con stack Nginx (reverse proxy) + Gunicorn + supervisión (systemd).
   - Certificados SSL vía Let’s Encrypt.
   - Configurar backups automáticos de la base de datos (pg_dump + cron).
9. **Monitoreo y mantenimiento**:
   - Logging estructurado en Django.
   - Scripts para generar reportes de cambios recientes.

## Modelado inicial de datos
- `User`: credenciales y datos básicos (email como identificador, nombre completo).
- `Profile` (OneToOne con `User`):
  - `parents_status`: choices ("ambos vivos", "uno vivo", "ninguno vivo").
  - `marital_status`: choices ("soltero", "casado", "separado", otros).
  - `children_count`: entero >= 0.
  - `job_title`, `company`, `job_summary` (texto libre, 500-1000 caracteres).
  - `phone_number` (opcional), `city`, `country`.
  - `updated_by` (último usuario) + `updated_at`.
  - Flags para consentimiento y visibilidad.
- `Invitation`:
  - `email`, `token`, `expires_at`, `is_used`, `approved`.
  - Flujo: admin genera invitación → usuario completa formulario inicial → admin aprueba → se activa cuenta.

## Flujos principales
1. **Alta inicial**: admin crea invitación o usuario → se envía enlace por WhatsApp → usuario establece contraseña y completa datos → se guarda perfil pendiente de aprobación → admin revisa en panel y aprueba.
2. **Edición**: usuario autenticado accede a su perfil → edita datos → se registra historial mediante auditoría.
3. **Exportación**: admin descarga CSV desde panel protegido.
4. **Baja**: usuario solicita baja → admin marca estado "de baja" y opcionalmente anonimiza datos (definir procedimiento legal). Mantener registro en historial.

## Próximos pasos sugeridos
1. Configurar proyecto Django base (`django-admin startproject`).
2. Definir modelos `Profile` y `Invitation`; aplicar migraciones.
3. Configurar autenticación y permisos (signals para crear `Profile` automáticamente).
4. Integrar auditoría (e.g., `django-simple-history`).
5. Construir vistas y formularios (basados en `class-based views`) para completar/editar perfil.
6. Implementar flujo de invitaciones y aprobación.
7. Añadir exportación a CSV y pruebas unitarias básicas.
8. Preparar scripts de despliegue y documentación de instalación.

## Consideraciones adicionales
- Documentar políticas de privacidad y consentimiento en la interfaz.
- Evaluar agregar captcha en formularios de auto-registro para evitar bots.
- Mantener repositorio privado y automatizar despliegue via CI/CD (opcional) conforme crezca.

Este plan establece las bases para iniciar el desarrollo de la aplicación con Django, asegurando privacidad, control de acceso y posibilidad de escalar las funcionalidades en el futuro.
