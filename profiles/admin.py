from django.contrib import admin

from .models import AlumniProfile, ProfileAuditLog


@admin.register(AlumniProfile)
class AlumniProfileAdmin(admin.ModelAdmin):
    list_display = (
        'user',
        'marital_status',
        'children_count',
        'company_name',
        'company_start_year',
        'consent_given',
        'updated_at',
    )
    list_filter = ('marital_status', 'parents_status', 'consent_given', 'company_start_year')
    search_fields = ('user__full_name', 'user__email')
    readonly_fields = ('created_at', 'updated_at', 'consent_timestamp')

    def save_model(self, request, obj, form, change):
        obj._updated_by = request.user
        super().save_model(request, obj, form, change)


@admin.register(ProfileAuditLog)
class ProfileAuditLogAdmin(admin.ModelAdmin):
    list_display = ('profile', 'changed_by', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('profile__user__full_name', 'changed_by__full_name')
    readonly_fields = ('profile', 'changed_by', 'changes', 'created_at')
